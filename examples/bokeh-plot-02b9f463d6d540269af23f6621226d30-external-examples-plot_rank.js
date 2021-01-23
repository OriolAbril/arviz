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
    
      
      
    
      var element = document.getElementById("d6c476cb-3996-4945-8f0e-c9abfce8bada");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'd6c476cb-3996-4945-8f0e-c9abfce8bada' but no matching script tag was found.")
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
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
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
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
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
                    
                  var docs_json = '{"058bf234-ece5-4fb0-85ef-06335130e572":{"roots":{"references":[{"attributes":{"axis":{"id":"39960"},"dimension":1,"ticker":null},"id":"39963","type":"Grid"},{"attributes":{"data_source":{"id":"40051"},"glyph":{"id":"40052"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"40053"},"selection_glyph":null,"view":{"id":"40055"}},"id":"40054","type":"GlyphRenderer"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#328c06"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40059","type":"VBar"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"39964"},{"id":"39965"},{"id":"39966"},{"id":"39967"},{"id":"39968"},{"id":"39969"},{"id":"39970"},{"id":"39971"}]},"id":"39974","type":"Toolbar"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"40007","type":"PolyAnnotation"},{"attributes":{},"id":"40077","type":"UnionRenderers"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"40074"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"39957"}},"id":"39956","type":"LinearAxis"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"39972","type":"BoxAnnotation"},{"attributes":{},"id":"40078","type":"Selection"},{"attributes":{"ticks":[0,1,2,3]},"id":"40041","type":"FixedTicker"},{"attributes":{"source":{"id":"40051"}},"id":"40055","type":"CDSView"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"39973","type":"PolyAnnotation"},{"attributes":{},"id":"40079","type":"UnionRenderers"},{"attributes":{},"id":"40080","type":"Selection"},{"attributes":{},"id":"40089","type":"BasicTickFormatter"},{"attributes":{"below":[{"id":"39990"}],"center":[{"id":"39993"},{"id":"39997"},{"id":"40050"},{"id":"40056"},{"id":"40062"},{"id":"40068"}],"left":[{"id":"39994"}],"output_backend":"webgl","plot_height":331,"plot_width":496,"renderers":[{"id":"40048"},{"id":"40054"},{"id":"40060"},{"id":"40066"}],"title":{"id":"40071"},"toolbar":{"id":"40008"},"toolbar_location":null,"x_range":{"id":"39948"},"x_scale":{"id":"39986"},"y_range":{"id":"39950"},"y_scale":{"id":"39988"}},"id":"39983","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"39952","type":"LinearScale"},{"attributes":{"children":[[{"id":"39947"},0,0],[{"id":"39983"},0,1]]},"id":"40103","type":"GridBox"},{"attributes":{"axis":{"id":"39956"},"ticker":null},"id":"39959","type":"Grid"},{"attributes":{"data":{"top":{"__ndarray__":"ZWZmZmZm7j9OG+i0gU7XP2cDnTbQad8/WfKLJb9Y2j9Bpw102kDTP17JL5b8Yt0/PW2g0wY60T9U8oslv1jaP1ws+cWSX9w/WlVVVVVV2T9SVVVVVVXZPzTQaQOdNuA/ZgOdNtBp3z9m8oslv1jaP0h+seQXS9Y/SH6x5BdL1j84baDTBjrRPz+nDXTaQNM/SH6x5BdL1j8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"40078"},"selection_policy":{"id":"40077"}},"id":"40017","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"40035"},"glyph":{"id":"40036"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"40037"},"selection_glyph":null,"view":{"id":"40039"}},"id":"40038","type":"GlyphRenderer"},{"attributes":{},"id":"40081","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"40017"},"glyph":{"id":"40018"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"40019"},"selection_glyph":null,"view":{"id":"40021"}},"id":"40020","type":"GlyphRenderer"},{"attributes":{},"id":"40082","type":"Selection"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40019","type":"VBar"},{"attributes":{},"id":"39954","type":"LinearScale"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40025","type":"VBar"},{"attributes":{"line_dash":[6],"location":1.4166666666666665},"id":"40028","type":"Span"},{"attributes":{"line_dash":[6],"location":3.480769230769231},"id":"40068","type":"Span"},{"attributes":{"children":[{"id":"40105"},{"id":"40103"}]},"id":"40106","type":"Column"},{"attributes":{"source":{"id":"40017"}},"id":"40021","type":"CDSView"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAA8D8OdNpApw30PxSuR+F6FPY/1AY6baDT9T8c6LSBThv4PxdLfrHkF/c/1QY6baDT9T+V/GLJL5b2P1jyiyW/WPc/43oUrkfh+T8ehetRuB75PxdLfrHkF/c/mJmZmZmZ9z8YrkfhehT2P1RVVVVVVfY/lfxiyS+W9j/gehSuR+H5P5iZmZmZmfc/kl8s+cWS9T8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"40080"},"selection_policy":{"id":"40079"}},"id":"40023","type":"ColumnDataSource"},{"attributes":{"axis_label":"Chain","formatter":{"id":"40076"},"ticker":{"id":"40041"}},"id":"39960","type":"LinearAxis"},{"attributes":{},"id":"40091","type":"BasicTickFormatter"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#fa7c17"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40024","type":"VBar"},{"attributes":{"source":{"id":"40057"}},"id":"40061","type":"CDSView"},{"attributes":{"text":"tau"},"id":"40043","type":"Title"},{"attributes":{"callback":null},"id":"39971","type":"HoverTool"},{"attributes":{"source":{"id":"40023"}},"id":"40027","type":"CDSView"},{"attributes":{},"id":"40092","type":"UnionRenderers"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#328c06"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40031","type":"VBar"},{"attributes":{"axis":{"id":"39990"},"ticker":null},"id":"39993","type":"Grid"},{"attributes":{},"id":"40093","type":"Selection"},{"attributes":{"callback":null},"id":"40005","type":"HoverTool"},{"attributes":{"line_dash":[6],"location":2.4166666666666665},"id":"40034","type":"Span"},{"attributes":{},"id":"39986","type":"LinearScale"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#328c06"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40030","type":"VBar"},{"attributes":{"axis_label":"Chain","formatter":{"id":"40091"},"ticker":{"id":"40069"}},"id":"39994","type":"LinearAxis"},{"attributes":{},"id":"40083","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"40023"},"glyph":{"id":"40024"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"40025"},"selection_glyph":null,"view":{"id":"40027"}},"id":"40026","type":"GlyphRenderer"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"40089"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"39991"}},"id":"39990","type":"LinearAxis"},{"attributes":{},"id":"40084","type":"Selection"},{"attributes":{},"id":"39991","type":"BasicTicker"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAAAECkcD0K1yMBQCa/WPKLpQFA6LSBThtoAkBqA5020OkCQCz5xZJfrANA8O7u7u5uBEDrUbgehWsDQOtRuB6FawNAC9ejcD0KA0BKfrHkF0sDQA102kCnDQRAqqqqqqoqA0BQG+i0gU4EQC+W/GLJrwRAThvotIFOBECuR+F6FC4EQE4b6LSBTgRAcD0K16PwBEA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"40082"},"selection_policy":{"id":"40081"}},"id":"40029","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"39973"}},"id":"39968","type":"LassoSelectTool"},{"attributes":{},"id":"39965","type":"PanTool"},{"attributes":{"source":{"id":"40029"}},"id":"40033","type":"CDSView"},{"attributes":{"axis":{"id":"39994"},"dimension":1,"ticker":null},"id":"39997","type":"Grid"},{"attributes":{"text":"mu"},"id":"40071","type":"Title"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#c10c90"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40036","type":"VBar"},{"attributes":{"data_source":{"id":"40029"},"glyph":{"id":"40030"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"40031"},"selection_glyph":null,"view":{"id":"40033"}},"id":"40032","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40018","type":"VBar"},{"attributes":{"overlay":{"id":"40006"}},"id":"40000","type":"BoxZoomTool"},{"attributes":{"line_dash":[6],"location":3.4166666666666665},"id":"40040","type":"Span"},{"attributes":{},"id":"39999","type":"PanTool"},{"attributes":{"data":{"top":{"__ndarray__":"MzMzMzOzDUC4HoXrUTgPQDCW/GLJrwxA0GkDnTbQDEBtoNMGOu0LQOi0gU4baApAzszMzMzMC0DrUbgehWsLQClcj8L1qApAqA102kAnCkBH4XoUrkcKQMaSXyz5xQlA6LSBThtoCkAqXI/C9agKQClcj8L1qApACtejcD0KC0AGOm2g0wYKQMkvlvxiyQpAaQOdNtDpCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"40084"},"selection_policy":{"id":"40083"}},"id":"40035","type":"ColumnDataSource"},{"attributes":{"toolbar":{"id":"40104"},"toolbar_location":"above"},"id":"40105","type":"ToolbarBox"},{"attributes":{},"id":"39998","type":"ResetTool"},{"attributes":{"below":[{"id":"39956"}],"center":[{"id":"39959"},{"id":"39963"},{"id":"40022"},{"id":"40028"},{"id":"40034"},{"id":"40040"}],"left":[{"id":"39960"}],"output_backend":"webgl","plot_height":331,"plot_width":496,"renderers":[{"id":"40020"},{"id":"40026"},{"id":"40032"},{"id":"40038"}],"title":{"id":"40043"},"toolbar":{"id":"39974"},"toolbar_location":null,"x_range":{"id":"39948"},"x_scale":{"id":"39952"},"y_range":{"id":"39950"},"y_scale":{"id":"39954"}},"id":"39947","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"40094","type":"UnionRenderers"},{"attributes":{},"id":"40004","type":"SaveTool"},{"attributes":{},"id":"39950","type":"DataRange1d"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#c10c90"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40064","type":"VBar"},{"attributes":{},"id":"40001","type":"WheelZoomTool"},{"attributes":{},"id":"40095","type":"Selection"},{"attributes":{},"id":"39964","type":"ResetTool"},{"attributes":{"overlay":{"id":"40007"}},"id":"40002","type":"LassoSelectTool"},{"attributes":{},"id":"40003","type":"UndoTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"39998"},{"id":"39999"},{"id":"40000"},{"id":"40001"},{"id":"40002"},{"id":"40003"},{"id":"40004"},{"id":"40005"}]},"id":"40008","type":"Toolbar"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#328c06"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40058","type":"VBar"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#c10c90"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40037","type":"VBar"},{"attributes":{"line_dash":[6],"location":1.4807692307692308},"id":"40056","type":"Span"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#fa7c17"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40052","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"6YVe6IVe4D9nZmZmZmbeP2dmZmZmZu4/WWqlVmql7D/eyI3cyI3YP7vQC73QC9U/uBM7sRM73T+vEzuxEzvdPyZ2Yid2Ytc/lxu5kRu52T8ZuZEbuZHfP5AbuZEbudk/QS/0Qi/04D8LwQ/8wA/cP5AbuZEbudk/q9RKrdRK4z9BL/RCL/TgPyZ2Yid2Ytc/USu1Uiu1wj8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"40093"},"selection_policy":{"id":"40092"}},"id":"40045","type":"ColumnDataSource"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40053","type":"VBar"},{"attributes":{"line_dash":[6],"location":0.41666666666666663},"id":"40022","type":"Span"},{"attributes":{"source":{"id":"40045"}},"id":"40049","type":"CDSView"},{"attributes":{},"id":"39948","type":"DataRange1d"},{"attributes":{},"id":"40074","type":"BasicTickFormatter"},{"attributes":{},"id":"40096","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"40057"},"glyph":{"id":"40058"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"40059"},"selection_glyph":null,"view":{"id":"40061"}},"id":"40060","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40046","type":"VBar"},{"attributes":{},"id":"40097","type":"Selection"},{"attributes":{"data":{"top":{"__ndarray__":"P/ADP/AD9z+SG7mRG7n2Py/0Qi/0QvU/eqEXeqEX9D9IbuRGbuT3P4If+IEf+PQ/MPRCL/RC9T+ZmZmZmZn3Pyd2Yid2YvQ/9kIv9EIv+D+4kRu5kRv7P7ATO7ETO/o/oBd6oRd6+D+mF3qhF3r4P1ZqpVZqpfk/9EIv9EIv+D9GbuRGbuT3P07sxE7sxPg/wA/8wA/8+z8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"40095"},"selection_policy":{"id":"40094"}},"id":"40051","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"39972"}},"id":"39966","type":"BoxZoomTool"},{"attributes":{"data":{"top":{"__ndarray__":"4Qd+4Af+BUDVSq3USq0EQBh6oRd6oQJAGHqhF3qhAkAUO7ETOzECQCu1Uiu10gRAd2IndmKnA0DFTuzETuwCQHIjN3IjNwNAJDdyIzfyA0Bu5EZu5MYCQB/4gR/4gQNAxU7sxE7sAkDTC73QCz0EQNALvdALPQRA0Au90As9BEB6oRd6oRcEQIIf+IEf+ARAhl7ohV5oBUA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"40097"},"selection_policy":{"id":"40096"}},"id":"40057","type":"ColumnDataSource"},{"attributes":{},"id":"39957","type":"BasicTicker"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40047","type":"VBar"},{"attributes":{"source":{"id":"40035"}},"id":"40039","type":"CDSView"},{"attributes":{},"id":"39969","type":"UndoTool"},{"attributes":{"source":{"id":"40063"}},"id":"40067","type":"CDSView"},{"attributes":{"data_source":{"id":"40045"},"glyph":{"id":"40046"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"40047"},"selection_glyph":null,"view":{"id":"40049"}},"id":"40048","type":"GlyphRenderer"},{"attributes":{},"id":"39967","type":"WheelZoomTool"},{"attributes":{"line_dash":[6],"location":2.480769230769231},"id":"40062","type":"Span"},{"attributes":{"data":{"top":{"__ndarray__":"EPzAD/zACUAg+IEf+IELQMEP/MAPfApAdmIndmKnC0A4ciM3ciMOQIZe6IVeaA1Ah17ohV5oDUDYiZ3YiR0NQD7wAz/wAw9Ae6EXeqEXDEAbuZEbuRELQHZiJ3ZipwtAeqEXeqEXDEB0IzdyIzcLQBu5kRu5EQtAFDuxEzsxCkByIzdyIzcLQBu5kRu5EQtAxU7sxE7sCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"40099"},"selection_policy":{"id":"40098"}},"id":"40063","type":"ColumnDataSource"},{"attributes":{"ticks":[0,1,2,3]},"id":"40069","type":"FixedTicker"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"40006","type":"BoxAnnotation"},{"attributes":{"line_dash":[6],"location":0.48076923076923067},"id":"40050","type":"Span"},{"attributes":{"toolbars":[{"id":"39974"},{"id":"40008"}],"tools":[{"id":"39964"},{"id":"39965"},{"id":"39966"},{"id":"39967"},{"id":"39968"},{"id":"39969"},{"id":"39970"},{"id":"39971"},{"id":"39998"},{"id":"39999"},{"id":"40000"},{"id":"40001"},{"id":"40002"},{"id":"40003"},{"id":"40004"},{"id":"40005"}]},"id":"40104","type":"ProxyToolbar"},{"attributes":{},"id":"39970","type":"SaveTool"},{"attributes":{},"id":"40098","type":"UnionRenderers"},{"attributes":{},"id":"40099","type":"Selection"},{"attributes":{},"id":"39988","type":"LinearScale"},{"attributes":{"data_source":{"id":"40063"},"glyph":{"id":"40064"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"40065"},"selection_glyph":null,"view":{"id":"40067"}},"id":"40066","type":"GlyphRenderer"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#c10c90"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"40065","type":"VBar"},{"attributes":{},"id":"40076","type":"BasicTickFormatter"}],"root_ids":["40106"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"058bf234-ece5-4fb0-85ef-06335130e572","root_ids":["40106"],"roots":{"40106":"d6c476cb-3996-4945-8f0e-c9abfce8bada"}}];
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