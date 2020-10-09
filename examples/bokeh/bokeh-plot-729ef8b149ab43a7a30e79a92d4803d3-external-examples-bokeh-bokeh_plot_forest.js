(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("1bdb8aaf-68e1-4cb4-b194-a6b3f44c716c");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '1bdb8aaf-68e1-4cb4-b194-a6b3f44c716c' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"d111a4d3-bdc8-46e7-9910-0769474a58ee":{"roots":{"references":[{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68459","type":"Line"},{"attributes":{"source":{"id":"68457"}},"id":"68461","type":"CDSView"},{"attributes":{"source":{"id":"68407"}},"id":"68411","type":"CDSView"},{"attributes":{"data":{"x":[0.9649316231388204,5.949680273009057],"y":[1.65,1.65]},"selected":{"id":"68503"},"selection_policy":{"id":"68504"}},"id":"68412","type":"ColumnDataSource"},{"attributes":{"source":{"id":"68372"}},"id":"68376","type":"CDSView"},{"attributes":{},"id":"68519","type":"Selection"},{"attributes":{},"id":"68517","type":"Selection"},{"attributes":{},"id":"68523","type":"Selection"},{"attributes":{"data_source":{"id":"68422"},"glyph":{"id":"68423"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68424"},"selection_glyph":null,"view":{"id":"68426"}},"id":"68425","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.181352887007439},"y":{"value":2.25}},"id":"68448","type":"Circle"},{"attributes":{},"id":"68509","type":"Selection"},{"attributes":{"data":{},"selected":{"id":"68517"},"selection_policy":{"id":"68518"}},"id":"68447","type":"ColumnDataSource"},{"attributes":{},"id":"68522","type":"UnionRenderers"},{"attributes":{"data":{"x":[2.393576367885318,6.880909766712211],"y":[1.95,1.95]},"selected":{"id":"68509"},"selection_policy":{"id":"68510"}},"id":"68427","type":"ColumnDataSource"},{"attributes":{},"id":"68502","type":"UnionRenderers"},{"attributes":{"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68428","type":"Line"},{"attributes":{},"id":"68505","type":"Selection"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68409","type":"Line"},{"attributes":{},"id":"68520","type":"UnionRenderers"},{"attributes":{"data":{"x":[-2.13476994033137,10.008689721107494],"y":[0.3,0.3]},"selected":{"id":"68483"},"selection_policy":{"id":"68484"}},"id":"68362","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68444","type":"Line"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.181352887007439},"y":{"value":2.25}},"id":"68449","type":"Circle"},{"attributes":{},"id":"68513","type":"Selection"},{"attributes":{},"id":"68518","type":"UnionRenderers"},{"attributes":{"data":{},"selected":{"id":"68511"},"selection_policy":{"id":"68512"}},"id":"68432","type":"ColumnDataSource"},{"attributes":{"source":{"id":"68437"}},"id":"68441","type":"CDSView"},{"attributes":{},"id":"68514","type":"UnionRenderers"},{"attributes":{},"id":"68507","type":"Selection"},{"attributes":{"data_source":{"id":"68442"},"glyph":{"id":"68443"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68444"},"selection_glyph":null,"view":{"id":"68446"}},"id":"68445","type":"GlyphRenderer"},{"attributes":{"source":{"id":"68442"}},"id":"68446","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68439","type":"Line"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.374380781729923},"y":{"value":1.65}},"id":"68418","type":"Circle"},{"attributes":{},"id":"68501","type":"Selection"},{"attributes":{},"id":"68506","type":"UnionRenderers"},{"attributes":{"source":{"id":"68462"}},"id":"68466","type":"CDSView"},{"attributes":{"data_source":{"id":"68452"},"glyph":{"id":"68453"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68454"},"selection_glyph":null,"view":{"id":"68456"}},"id":"68455","type":"GlyphRenderer"},{"attributes":{"data":{},"selected":{"id":"68505"},"selection_policy":{"id":"68506"}},"id":"68417","type":"ColumnDataSource"},{"attributes":{"source":{"id":"68422"}},"id":"68426","type":"CDSView"},{"attributes":{"source":{"id":"68397"}},"id":"68401","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68429","type":"Line"},{"attributes":{"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68443","type":"Line"},{"attributes":{"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68413","type":"Line"},{"attributes":{},"id":"68515","type":"Selection"},{"attributes":{},"id":"68504","type":"UnionRenderers"},{"attributes":{"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68423","type":"Line"},{"attributes":{},"id":"68524","type":"UnionRenderers"},{"attributes":{},"id":"68521","type":"Selection"},{"attributes":{"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68398","type":"Line"},{"attributes":{"data_source":{"id":"68357"},"glyph":{"id":"68358"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68359"},"selection_glyph":null,"view":{"id":"68361"}},"id":"68360","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.74129666559688},"y":{"value":1.95}},"id":"68433","type":"Circle"},{"attributes":{"source":{"id":"68432"}},"id":"68436","type":"CDSView"},{"attributes":{"source":{"id":"68427"}},"id":"68431","type":"CDSView"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.74129666559688},"y":{"value":1.95}},"id":"68434","type":"Circle"},{"attributes":{"data_source":{"id":"68427"},"glyph":{"id":"68428"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68429"},"selection_glyph":null,"view":{"id":"68431"}},"id":"68430","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"68462"},"glyph":{"id":"68463"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68464"},"selection_glyph":null,"view":{"id":"68466"}},"id":"68465","type":"GlyphRenderer"},{"attributes":{},"id":"68510","type":"UnionRenderers"},{"attributes":{"data":{"x":[2.2921199846409115,6.479187946875487],"y":[0.8999999999999999,0.8999999999999999]},"selected":{"id":"68497"},"selection_policy":{"id":"68498"}},"id":"68397","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68414","type":"Line"},{"attributes":{"source":{"id":"68357"}},"id":"68361","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68424","type":"Line"},{"attributes":{},"id":"68516","type":"UnionRenderers"},{"attributes":{"below":[{"id":"68320"}],"center":[{"id":"68323"},{"id":"68327"}],"left":[{"id":"68324"}],"outline_line_color":null,"output_backend":"webgl","plot_height":435,"plot_width":360,"renderers":[{"id":"68350"},{"id":"68355"},{"id":"68360"},{"id":"68365"},{"id":"68370"},{"id":"68375"},{"id":"68380"},{"id":"68385"},{"id":"68390"},{"id":"68395"},{"id":"68400"},{"id":"68405"},{"id":"68410"},{"id":"68415"},{"id":"68420"},{"id":"68425"},{"id":"68430"},{"id":"68435"},{"id":"68440"},{"id":"68445"},{"id":"68450"},{"id":"68455"},{"id":"68460"},{"id":"68465"}],"title":{"id":"68467"},"toolbar":{"id":"68338"},"toolbar_location":null,"x_range":{"id":"68469"},"x_scale":{"id":"68316"},"y_range":{"id":"68470"},"y_scale":{"id":"68318"}},"id":"68311","subtype":"Figure","type":"Plot"},{"attributes":{"fill_color":{"value":"#fa7c17"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.3706630373472235},"y":{"value":0.8999999999999999}},"id":"68403","type":"Circle"},{"attributes":{"data_source":{"id":"68402"},"glyph":{"id":"68403"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68404"},"selection_glyph":null,"view":{"id":"68406"}},"id":"68405","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.374380781729923},"y":{"value":1.65}},"id":"68419","type":"Circle"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68399","type":"Line"},{"attributes":{"data_source":{"id":"68432"},"glyph":{"id":"68433"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68434"},"selection_glyph":null,"view":{"id":"68436"}},"id":"68435","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.3706630373472235},"y":{"value":0.8999999999999999}},"id":"68404","type":"Circle"},{"attributes":{"data":{},"selected":{"id":"68523"},"selection_policy":{"id":"68524"}},"id":"68462","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"68417"},"glyph":{"id":"68418"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68419"},"selection_glyph":null,"view":{"id":"68421"}},"id":"68420","type":"GlyphRenderer"},{"attributes":{"data":{},"selected":{"id":"68499"},"selection_policy":{"id":"68500"}},"id":"68402","type":"ColumnDataSource"},{"attributes":{},"id":"68512","type":"UnionRenderers"},{"attributes":{"data":{"x":[-0.7842313478998125,9.985046696913068],"y":[2.55,2.55]},"selected":{"id":"68519"},"selection_policy":{"id":"68520"}},"id":"68452","type":"ColumnDataSource"},{"attributes":{"ticks":[0.44999999999999996,2.0999999999999996]},"id":"68471","type":"FixedTicker"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.4162688471038556},"y":{"value":2.55}},"id":"68463","type":"Circle"},{"attributes":{"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68408","type":"Line"},{"attributes":{},"id":"68503","type":"Selection"},{"attributes":{"bounds":"auto","min_interval":1},"id":"68469","type":"DataRange1d"},{"attributes":{},"id":"68508","type":"UnionRenderers"},{"attributes":{"source":{"id":"68402"}},"id":"68406","type":"CDSView"},{"attributes":{"data":{"x":[-2.358040118461496,10.910290467635015],"y":[1.95,1.95]},"selected":{"id":"68507"},"selection_policy":{"id":"68508"}},"id":"68422","type":"ColumnDataSource"},{"attributes":{"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68458","type":"Line"},{"attributes":{"source":{"id":"68412"}},"id":"68416","type":"CDSView"},{"attributes":{"source":{"id":"68417"}},"id":"68421","type":"CDSView"},{"attributes":{"data_source":{"id":"68437"},"glyph":{"id":"68438"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68439"},"selection_glyph":null,"view":{"id":"68441"}},"id":"68440","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"68412"},"glyph":{"id":"68413"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68414"},"selection_glyph":null,"view":{"id":"68416"}},"id":"68415","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.663971371122512},"y":{"value":0}},"id":"68359","type":"Circle"},{"attributes":{"data":{"x":[1.9127415870254314,7.0411499662839026],"y":[2.25,2.25]},"selected":{"id":"68515"},"selection_policy":{"id":"68516"}},"id":"68442","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.4162688471038556},"y":{"value":2.55}},"id":"68464","type":"Circle"},{"attributes":{"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68438","type":"Line"},{"attributes":{"toolbars":[{"id":"68338"}],"tools":[{"id":"68328"},{"id":"68329"},{"id":"68330"},{"id":"68331"},{"id":"68332"},{"id":"68333"},{"id":"68334"},{"id":"68335"}]},"id":"68529","type":"ProxyToolbar"},{"attributes":{},"id":"68511","type":"Selection"},{"attributes":{"data":{"x":[2.454317217804662,6.859709821773684],"y":[0,0]},"selected":{"id":"68479"},"selection_policy":{"id":"68480"}},"id":"68352","type":"ColumnDataSource"},{"attributes":{"formatter":{"id":"68474"},"ticker":{"id":"68321"}},"id":"68320","type":"LinearAxis"},{"attributes":{"data_source":{"id":"68457"},"glyph":{"id":"68458"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68459"},"selection_glyph":null,"view":{"id":"68461"}},"id":"68460","type":"GlyphRenderer"},{"attributes":{},"id":"68485","type":"Selection"},{"attributes":{},"id":"68482","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.59253463805363},"y":{"value":0.6}},"id":"68389","type":"Circle"},{"attributes":{},"id":"68486","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68369","type":"Line"},{"attributes":{"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68368","type":"Line"},{"attributes":{},"id":"68321","type":"BasicTicker"},{"attributes":{"data":{"x":[2.2808674873161356,6.6262125124574265],"y":[0.3,0.3]},"selected":{"id":"68485"},"selection_policy":{"id":"68486"}},"id":"68367","type":"ColumnDataSource"},{"attributes":{},"id":"68477","type":"Selection"},{"attributes":{},"id":"68494","type":"UnionRenderers"},{"attributes":{},"id":"68480","type":"UnionRenderers"},{"attributes":{},"id":"68491","type":"Selection"},{"attributes":{"data":{},"selected":{"id":"68493"},"selection_policy":{"id":"68494"}},"id":"68387","type":"ColumnDataSource"},{"attributes":{},"id":"68328","type":"ResetTool"},{"attributes":{},"id":"68487","type":"Selection"},{"attributes":{},"id":"68499","type":"Selection"},{"attributes":{"source":{"id":"68392"}},"id":"68396","type":"CDSView"},{"attributes":{},"id":"68489","type":"Selection"},{"attributes":{},"id":"68334","type":"SaveTool"},{"attributes":{"data_source":{"id":"68397"},"glyph":{"id":"68398"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68399"},"selection_glyph":null,"view":{"id":"68401"}},"id":"68400","type":"GlyphRenderer"},{"attributes":{"children":[{"id":"68530"},{"id":"68528"}]},"id":"68531","type":"Column"},{"attributes":{"source":{"id":"68377"}},"id":"68381","type":"CDSView"},{"attributes":{"fill_color":{"value":"#fa7c17"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.409527442378456},"y":{"value":0.3}},"id":"68373","type":"Circle"},{"attributes":{"fill_color":{"value":"#fa7c17"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.59253463805363},"y":{"value":0.6}},"id":"68388","type":"Circle"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68384","type":"Line"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"68336","type":"BoxAnnotation"},{"attributes":{},"id":"68498","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68454","type":"Line"},{"attributes":{"axis":{"id":"68324"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"68327","type":"Grid"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68354","type":"Line"},{"attributes":{"data":{"x":[-2.061896746944455,11.515205993744095],"y":[2.25,2.25]},"selected":{"id":"68513"},"selection_policy":{"id":"68514"}},"id":"68437","type":"ColumnDataSource"},{"attributes":{"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68348","type":"Line"},{"attributes":{"source":{"id":"68382"}},"id":"68386","type":"CDSView"},{"attributes":{"data_source":{"id":"68372"},"glyph":{"id":"68373"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68374"},"selection_glyph":null,"view":{"id":"68376"}},"id":"68375","type":"GlyphRenderer"},{"attributes":{"text":"94.0% HDI"},"id":"68467","type":"Title"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68394","type":"Line"},{"attributes":{},"id":"68331","type":"WheelZoomTool"},{"attributes":{"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68363","type":"Line"},{"attributes":{"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68393","type":"Line"},{"attributes":{"data":{"x":[-1.9958943490877263,9.311550558002505],"y":[1.65,1.65]},"selected":{"id":"68501"},"selection_policy":{"id":"68502"}},"id":"68407","type":"ColumnDataSource"},{"attributes":{},"id":"68481","type":"Selection"},{"attributes":{"data_source":{"id":"68367"},"glyph":{"id":"68368"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68369"},"selection_glyph":null,"view":{"id":"68371"}},"id":"68370","type":"GlyphRenderer"},{"attributes":{"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68383","type":"Line"},{"attributes":{},"id":"68479","type":"Selection"},{"attributes":{"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68378","type":"Line"},{"attributes":{"data_source":{"id":"68377"},"glyph":{"id":"68378"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68379"},"selection_glyph":null,"view":{"id":"68381"}},"id":"68380","type":"GlyphRenderer"},{"attributes":{},"id":"68493","type":"Selection"},{"attributes":{"data_source":{"id":"68387"},"glyph":{"id":"68388"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68389"},"selection_glyph":null,"view":{"id":"68391"}},"id":"68390","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"#fa7c17"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.663971371122512},"y":{"value":0}},"id":"68358","type":"Circle"},{"attributes":{"data":{},"selected":{"id":"68487"},"selection_policy":{"id":"68488"}},"id":"68372","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"68320"},"ticker":null},"id":"68323","type":"Grid"},{"attributes":{"data":{"x":[-1.0798320889339172,10.200853218312133],"y":[0,0]},"selected":{"id":"68477"},"selection_policy":{"id":"68478"}},"id":"68347","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"68352"},"glyph":{"id":"68353"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68354"},"selection_glyph":null,"view":{"id":"68356"}},"id":"68355","type":"GlyphRenderer"},{"attributes":{"data":{},"selected":{"id":"68481"},"selection_policy":{"id":"68482"}},"id":"68357","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"68447"},"glyph":{"id":"68448"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68449"},"selection_glyph":null,"view":{"id":"68451"}},"id":"68450","type":"GlyphRenderer"},{"attributes":{"data":{"x":[-2.2641273152868076,10.475301593619335],"y":[0.8999999999999999,0.8999999999999999]},"selected":{"id":"68495"},"selection_policy":{"id":"68496"}},"id":"68392","type":"ColumnDataSource"},{"attributes":{},"id":"68492","type":"UnionRenderers"},{"attributes":{"source":{"id":"68362"}},"id":"68366","type":"CDSView"},{"attributes":{"data_source":{"id":"68382"},"glyph":{"id":"68383"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68384"},"selection_glyph":null,"view":{"id":"68386"}},"id":"68385","type":"GlyphRenderer"},{"attributes":{},"id":"68333","type":"UndoTool"},{"attributes":{"data_source":{"id":"68407"},"glyph":{"id":"68408"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68409"},"selection_glyph":null,"view":{"id":"68411"}},"id":"68410","type":"GlyphRenderer"},{"attributes":{},"id":"68495","type":"Selection"},{"attributes":{"source":{"id":"68352"}},"id":"68356","type":"CDSView"},{"attributes":{},"id":"68478","type":"UnionRenderers"},{"attributes":{},"id":"68329","type":"PanTool"},{"attributes":{"data":{"x":[2.1521098502407368,6.9643589964055215],"y":[0.6,0.6]},"selected":{"id":"68491"},"selection_policy":{"id":"68492"}},"id":"68382","type":"ColumnDataSource"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"68328"},{"id":"68329"},{"id":"68330"},{"id":"68331"},{"id":"68332"},{"id":"68333"},{"id":"68334"},{"id":"68335"}]},"id":"68338","type":"Toolbar"},{"attributes":{"overlay":{"id":"68337"}},"id":"68332","type":"LassoSelectTool"},{"attributes":{},"id":"68483","type":"Selection"},{"attributes":{"source":{"id":"68452"}},"id":"68456","type":"CDSView"},{"attributes":{"children":[[{"id":"68311"},0,0]]},"id":"68528","type":"GridBox"},{"attributes":{"data_source":{"id":"68347"},"glyph":{"id":"68348"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68349"},"selection_glyph":null,"view":{"id":"68351"}},"id":"68350","type":"GlyphRenderer"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"68337","type":"PolyAnnotation"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.409527442378456},"y":{"value":0.3}},"id":"68374","type":"Circle"},{"attributes":{},"id":"68476","type":"BasicTickFormatter"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68349","type":"Line"},{"attributes":{},"id":"68474","type":"BasicTickFormatter"},{"attributes":{},"id":"68497","type":"Selection"},{"attributes":{"overlay":{"id":"68336"}},"id":"68330","type":"BoxZoomTool"},{"attributes":{"source":{"id":"68367"}},"id":"68371","type":"CDSView"},{"attributes":{"toolbar":{"id":"68529"},"toolbar_location":"above"},"id":"68530","type":"ToolbarBox"},{"attributes":{"bounds":"auto","end":3.4499999999999997,"min_interval":2,"start":-0.8999999999999999},"id":"68470","type":"DataRange1d"},{"attributes":{"source":{"id":"68347"}},"id":"68351","type":"CDSView"},{"attributes":{"data":{"x":[1.9790083397469873,5.455951625836456],"y":[2.55,2.55]},"selected":{"id":"68521"},"selection_policy":{"id":"68522"}},"id":"68457","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"68392"},"glyph":{"id":"68393"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68394"},"selection_glyph":null,"view":{"id":"68396"}},"id":"68395","type":"GlyphRenderer"},{"attributes":{"data":{"x":[-1.4753067334040253,11.208804111875878],"y":[0.6,0.6]},"selected":{"id":"68489"},"selection_policy":{"id":"68490"}},"id":"68377","type":"ColumnDataSource"},{"attributes":{},"id":"68490","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68379","type":"Line"},{"attributes":{},"id":"68488","type":"UnionRenderers"},{"attributes":{},"id":"68500","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"68362"},"glyph":{"id":"68363"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68364"},"selection_glyph":null,"view":{"id":"68366"}},"id":"68365","type":"GlyphRenderer"},{"attributes":{"source":{"id":"68447"}},"id":"68451","type":"CDSView"},{"attributes":{"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"68353","type":"Line"},{"attributes":{},"id":"68496","type":"UnionRenderers"},{"attributes":{},"id":"68318","type":"LinearScale"},{"attributes":{"source":{"id":"68387"}},"id":"68391","type":"CDSView"},{"attributes":{},"id":"68316","type":"LinearScale"},{"attributes":{"formatter":{"id":"68476"},"major_label_overrides":{"0.44999999999999996":"Non Centered: mu","2.0999999999999996":"Centered: mu"},"ticker":{"id":"68471"}},"id":"68324","type":"LinearAxis"},{"attributes":{},"id":"68484","type":"UnionRenderers"},{"attributes":{"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68453","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"68364","type":"Line"},{"attributes":{"callback":null},"id":"68335","type":"HoverTool"}],"root_ids":["68531"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"d111a4d3-bdc8-46e7-9910-0769474a58ee","root_ids":["68531"],"roots":{"68531":"1bdb8aaf-68e1-4cb4-b194-a6b3f44c716c"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();